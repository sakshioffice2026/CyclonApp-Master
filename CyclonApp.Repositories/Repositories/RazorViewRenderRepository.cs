using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CyclonApp.Repositories.Contracts;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc.Abstractions;
using Microsoft.AspNetCore.Mvc.ModelBinding;
using Microsoft.AspNetCore.Mvc.Razor;
using Microsoft.AspNetCore.Mvc.Rendering;
using Microsoft.AspNetCore.Mvc.ViewFeatures;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Routing;

namespace CyclonApp.Repositories.Repositories
{
    public class RazorViewRenderer : IRazorViewRenderer
    {
        private readonly IRazorViewEngine _viewEngine;
        private readonly ITempDataProvider _tempDataProvider;
        private readonly IServiceProvider _serviceProvider;
        private readonly IHttpContextAccessor _httpContextAccessor;

        public RazorViewRenderer(
            IRazorViewEngine viewEngine,
            ITempDataProvider tempDataProvider,
            IServiceProvider serviceProvider,
            IHttpContextAccessor httpContextAccessor)
        {
            _viewEngine = viewEngine;
            _tempDataProvider = tempDataProvider;
            _serviceProvider = serviceProvider;
            _httpContextAccessor = httpContextAccessor;
        }

        public async Task<string> RenderViewAsync<TModel>(string viewName, TModel model)
        {
            var httpContext = _httpContextAccessor.HttpContext
                ?? new DefaultHttpContext { RequestServices = _serviceProvider };

            var actionContext = new ActionContext(
                httpContext,
                httpContext.GetRouteData() ?? new Microsoft.AspNetCore.Routing.RouteData(),
                new ActionDescriptor());

            await using var sw = new StringWriter();

            var viewResult = _viewEngine.FindView(actionContext, viewName, false);
            if (!viewResult.Success)
                viewResult = _viewEngine.GetView(null, viewName, false);

            if (!viewResult.Success)
                throw new InvalidOperationException($"Could not find view '{viewName}'.");

            var viewDictionary = new ViewDataDictionary<TModel>(
                new EmptyModelMetadataProvider(),
                new ModelStateDictionary())
            { Model = model };

            var tempData = new TempDataDictionary(httpContext, _tempDataProvider);

            var viewContext = new ViewContext(
                actionContext,
                viewResult.View,
                viewDictionary,
                tempData,
                sw,
                new HtmlHelperOptions());

            await viewResult.View.RenderAsync(viewContext);
            return sw.ToString();
        }
    }

}
